import {createTRPCRouter, protectedProcedure, publicProcedure} from "~/server/api/trpc";
import {z} from "zod";
import {eq} from "drizzle-orm";
import {users} from "~/server/db/schema";

export const userRouter = createTRPCRouter({
    // defines purpose, protected = requires login
    byId: publicProcedure
        // validates input
        .input(z.string())
        // sets up query with context and input
        .query(({ctx, input}) => {
            // drizzle orm select
            // https://orm.drizzle.team/docs/rqb
            return ctx.db.query.users.findFirst({
                where: eq(users.id, input),
              with: {
                  favorites: {
                    with: {
                      movie: true,
                    },
                  },
              }
            })
        }),

    getAll: publicProcedure.query(({ctx}) => {
        return ctx.db.query.users.findMany();
    }),
});