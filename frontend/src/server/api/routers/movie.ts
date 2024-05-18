import { z } from 'zod';
import { eq } from 'drizzle-orm';

import {
  createTRPCRouter,
  protectedProcedure,
  publicProcedure,
} from '~/server/api/trpc';
import { movies } from '~/server/db/schema';

//** movie route */
export const movieRouter = createTRPCRouter({
  // defines purpose
  byId: publicProcedure
    // validates input
    .input(z.string())
    // sets up query with context and input
    .query(async ({ctx, input}) => {
      // drizzle orm select
      // https://orm.drizzle.team/docs/rqb
      const movie = await ctx.db.query.movies.findFirst({
        with:{
          id: input
        },
      });
      return movie;
  }),

  getMovie: protectedProcedure
    .input(z.object({ id: z.number() }))
    .query(({ ctx, input }) => {
      return ctx.db.query.movies.findFirst({
        where: eq(movies.id, input.id),
      })
    }),

  getAll: protectedProcedure.query(({ ctx }) => {
    return ctx.db.query.movies.findMany();
  }),
});
