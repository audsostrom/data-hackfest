import { z } from 'zod';
import { eq } from 'drizzle-orm';

import {
  createTRPCRouter,
  protectedProcedure,
  publicProcedure,
} from '~/server/api/trpc';
import { movieProductions, movies } from '~/server/db/schema';

//** movie route */
export const movieRouter = createTRPCRouter({
  // defines purpose, protected = requires login
  byId: protectedProcedure
    // validates input
    .input(z.number())
    // sets up query with context and input
    .query(({ ctx, input }) => {
      // drizzle orm select
      // https://orm.drizzle.team/docs/rqb
      return ctx.db.query.movies.findFirst({
        where: eq(movies.id, input),
        with:{
          movieProductions: {
            with:{
              production: true,
            },
          },
        }
      })
    }),

  getAll: protectedProcedure.query(({ ctx }) => {
    return ctx.db.query.movies.findMany();
  }),
});
